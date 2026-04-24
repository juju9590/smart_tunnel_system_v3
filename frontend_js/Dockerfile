FROM node:22 AS build-stage
WORKDIR /app
COPY package*.json ./
RUN npm install
ARG VITE_KAKAO_MAP_API_KEY=de202e5f1e1b8c26cb092fb674de768d
ENV VITE_KAKAO_MAP_API_KEY=$VITE_KAKAO_MAP_API_KEY

COPY . .
RUN npm run build

FROM nginx:stable-alpine
COPY --from=build-stage /app/dist /usr/share/nginx/html
# React Router 경로 인식을 위한 설정 (4단계 참고)
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]